"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { AgentModeBadge, StatusBadge } from "@/components/panels/Primitives";
import { api, ApiError } from "@/lib/api-client";
import type { ExperimentRead, LearningProgressRead, MetricsRead } from "@/types/api";

export default function DashboardPage() {
  const [experiments, setExperiments] = useState<ExperimentRead[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [featured, setFeatured] = useState<ExperimentRead | null>(null);
  const [metrics, setMetrics] = useState<MetricsRead | null>(null);
  const [lpHistory, setLpHistory] = useState<LearningProgressRead[]>([]);

  useEffect(() => {
    let cancelled = false;
    api
      .listExperiments()
      .then((list) => {
        if (cancelled) return;
        setExperiments(list);
        const latest = [...list].sort((a, b) => b.id - a.id)[0] ?? null;
        setFeatured(latest);
      })
      .catch((e) => setError(e instanceof ApiError ? `API error (${e.status})` : "Failed to load experiments"));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!featured) return;
    let cancelled = false;
    api.getMetrics(featured.id).then((m) => !cancelled && setMetrics(m));
    api.getLearningProgress(featured.id).then((lp) => !cancelled && setLpHistory(lp));
    return () => {
      cancelled = true;
    };
  }, [featured]);

  const TERMINAL_STATUSES = new Set(["completed", "failed", "stopped"]);
  const active = experiments?.filter((e) => !TERMINAL_STATUSES.has(e.status)).length ?? 0;
  const completed = experiments?.filter((e) => e.status === "completed").length ?? 0;

  const chartData = lpHistory.map((r, i) => ({
    step: i,
    successRate: r.performance_after,
    learningProgress: r.learning_progress,
  }));

  const featuredHasAgent = featured?.agent_mode === "demo_ppo";

  return (
    <Shell
      title="LAB OVERVIEW"
      subtitle="AI Environment Generator for Training Other AI"
      actions={
        <Link
          href="/experiments/new"
          className="px-3 py-1.5 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea]"
        >
          New Experiment
        </Link>
      }
    >
      {error && <div className="text-app-red text-xs">{error}</div>}

      <div className="grid grid-cols-5 gap-4">
        <Panel title="ACTIVE EXPERIMENTS">
          <div className="text-2xl font-bold text-app-blue">{active}</div>
        </Panel>
        <Panel title="COMPLETED EXPERIMENTS">
          <div className="text-2xl font-bold text-app-green">{completed}</div>
        </Panel>
        <Panel title="LATEST SUCCESS RATE">
          <div className="text-2xl font-bold text-app-hi">
            {featuredHasAgent && metrics?.latest_success_rate != null ? metrics.latest_success_rate.toFixed(2) : "--"}
          </div>
          {!featuredHasAgent && <div className="text-[10px] text-app-low mt-1">No agent connected</div>}
        </Panel>
        <Panel title="CURRENT WEAKNESS">
          <div className="text-lg font-bold text-app-orange truncate">
            {featuredHasAgent ? (metrics?.latest_primary_weakness ?? "--") : "--"}
          </div>
          {!featuredHasAgent && <div className="text-[10px] text-app-low mt-1">No agent connected</div>}
        </Panel>
        <Panel title="LEARNING PROGRESS">
          <div
            className={`text-2xl font-bold ${
              (metrics?.latest_learning_progress ?? 0) >= 0 ? "text-app-green" : "text-app-red"
            }`}
          >
            {featuredHasAgent && metrics?.latest_learning_progress != null
              ? `${metrics.latest_learning_progress >= 0 ? "+" : ""}${metrics.latest_learning_progress.toFixed(3)}`
              : "--"}
          </div>
          {!featuredHasAgent && <div className="text-[10px] text-app-low mt-1">No agent connected</div>}
        </Panel>
      </div>

      {featuredHasAgent ? (
        <div className="grid grid-cols-2 gap-4">
          <Panel title="SUCCESS RATE vs ENVIRONMENTS" className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="#1a2236" strokeDasharray="2 4" />
                <XAxis dataKey="step" stroke="#566078" fontSize={10} />
                <YAxis stroke="#566078" fontSize={10} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: "#0f1524", border: "1px solid #1f2a40", fontSize: 11 }} />
                <Line type="monotone" dataKey="successRate" stroke="#3f8cff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
          <Panel title="LEARNING PROGRESS vs ENVIRONMENTS" className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="#1a2236" strokeDasharray="2 4" />
                <XAxis dataKey="step" stroke="#566078" fontSize={10} />
                <YAxis stroke="#566078" fontSize={10} />
                <Tooltip contentStyle={{ background: "#0f1524", border: "1px solid #1f2a40", fontSize: 11 }} />
                <Line type="monotone" dataKey="learningProgress" stroke="#22c55e" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
        </div>
      ) : (
        featured && (
          <Panel title={featured.agent_mode === "external" ? "AGENT" : "TRAINING"}>
            <div className="text-xs text-app-mid">
              {featured.agent_mode === "external"
                ? "Waiting for connection. Success-rate and learning-progress charts require an agent."
                : "Environment generation does not require an agent. Success-rate and learning-progress charts only apply to Demo Agent -- PPO runs."}
            </div>
          </Panel>
        )
      )}

      <Panel title="EXPERIMENTS">
        {!experiments ? (
          <div className="text-app-mid text-xs">Loading...</div>
        ) : experiments.length === 0 ? (
          <div className="text-app-mid text-xs">
            No experiments yet.{" "}
            <Link href="/experiments/new" className="text-app-blue">
              Create one
            </Link>
            .
          </div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-app-low">
                <th className="pb-2 font-medium">ID</th>
                <th className="pb-2 font-medium">Name</th>
                <th className="pb-2 font-medium">Agent</th>
                <th className="pb-2 font-medium">Curriculum</th>
                <th className="pb-2 font-medium">Seed</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {experiments
                .slice()
                .sort((a, b) => b.id - a.id)
                .map((e) => (
                  <tr key={e.id} className="border-t border-[#1a2033] hover:bg-app-panel2">
                    <td className="py-1.5 font-mono text-app-mid">{e.id}</td>
                    <td className="py-1.5">
                      <Link href={`/experiments/${e.id}`} className="text-app-blue hover:underline">
                        {e.name}
                      </Link>
                    </td>
                    <td className="py-1.5">
                      <AgentModeBadge agentMode={e.agent_mode} />
                    </td>
                    <td className="py-1.5 text-app-hi">{e.curriculum_type}</td>
                    <td className="py-1.5 font-mono text-app-mid">{e.seed}</td>
                    <td className="py-1.5">
                      <StatusBadge status={e.status} />
                    </td>
                    <td className="py-1.5 text-app-mid">{new Date(e.created_at).toLocaleString()}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </Panel>
    </Shell>
  );
}
