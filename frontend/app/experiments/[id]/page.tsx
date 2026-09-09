"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { AgentModeBadge, StatChip, StatusBadge } from "@/components/panels/Primitives";
import { api } from "@/lib/api-client";
import type {
  EnvironmentRead,
  ExperimentRead,
  GeneralizationRead,
  LearningProgressRead,
  WeaknessRead,
} from "@/types/api";

const TERMINAL_STATUSES = new Set(["completed", "failed", "stopped"]);

const POLL_INTERVAL_MS = 3000;

export default function ExperimentDetailsPage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [experiment, setExperiment] = useState<ExperimentRead | null>(null);
  const [environments, setEnvironments] = useState<EnvironmentRead[]>([]);
  const [weaknesses, setWeaknesses] = useState<WeaknessRead[]>([]);
  const [learningProgress, setLearningProgress] = useState<LearningProgressRead[]>([]);
  const [generalization, setGeneralization] = useState<GeneralizationRead[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const exp = await api.getExperiment(experimentId);
    setExperiment(exp);
    const [envs, weak, lp, gen] = await Promise.all([
      api.getEnvironments(experimentId),
      api.getWeaknesses(experimentId),
      api.getLearningProgress(experimentId),
      api.getGeneralization(experimentId),
    ]);
    setEnvironments(envs);
    setWeaknesses(weak);
    setLearningProgress(lp);
    setGeneralization(gen);
  }, [experimentId]);

  useEffect(() => {
    refresh().catch(() => setError("Failed to load experiment"));
  }, [refresh]);

  useEffect(() => {
    if (!experiment?.status || TERMINAL_STATUSES.has(experiment.status)) return;
    const interval = setInterval(() => refresh().catch(() => {}), POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [experiment?.status, refresh]);

  async function handleStart() {
    setBusy(true);
    setError(null);
    try {
      await api.startExperiment(experimentId);
      await refresh();
    } catch {
      setError("Failed to start experiment");
    } finally {
      setBusy(false);
    }
  }

  async function handleStop() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.stopExperiment(experimentId);
      if (res.message) setError(res.message);
      await refresh();
    } catch {
      setError("Failed to stop experiment");
    } finally {
      setBusy(false);
    }
  }

  if (!experiment) {
    return (
      <Shell title="EXPERIMENT" experimentId={experimentId}>
        <div className="text-app-mid text-xs">Loading...</div>
      </Shell>
    );
  }

  const isEnvOnly = experiment.agent_mode === "environment_only";
  const isExternal = experiment.agent_mode === "external";
  const isDemoPpo = experiment.agent_mode === "demo_ppo";
  const canStart = experiment.status === "pending";
  const canStop = !TERMINAL_STATUSES.has(experiment.status) && experiment.status !== "pending";

  return (
    <Shell
      experimentId={experimentId}
      title={experiment.name.toUpperCase()}
      subtitle={
        <span className="flex items-center gap-3">
          <StatusBadge status={experiment.status} />
          <AgentModeBadge agentMode={experiment.agent_mode} />
          {isDemoPpo && <span>Curriculum: {experiment.curriculum_type}</span>}
          <span>Seed: {experiment.seed}</span>
        </span>
      }
      actions={
        <div className="flex gap-2">
          {canStart && (
            <button
              onClick={handleStart}
              disabled={busy}
              className="px-3 py-1.5 rounded bg-app-blue text-white text-[12px] font-semibold hover:bg-[#2f78ea] disabled:opacity-50"
            >
              Start
            </button>
          )}
          {canStop && (
            <button
              onClick={handleStop}
              disabled={busy}
              className="px-3 py-1.5 rounded bg-[#3a1420] border border-[#5c2030] text-[#ff6b6b] text-[12px] font-semibold hover:bg-[#471926] disabled:opacity-50"
            >
              Stop
            </button>
          )}
          <Link
            href={`/live-training/${experimentId}`}
            className="px-3 py-1.5 rounded border border-app-border text-app-hi text-[12px] hover:bg-app-panel2"
          >
            {isEnvOnly ? "Live Generation" : isExternal ? "External Agent" : "Live Training"}
          </Link>
        </div>
      }
    >
      {error && <div className="text-app-orange text-xs">{error}</div>}

      <div className="grid grid-cols-4 gap-4">
        <StatChip label={isEnvOnly ? "Environments Generated" : "Environments"} value={environments.length} />
        {isDemoPpo && (
          <StatChip
            label="Latest Success Rate"
            value={
              learningProgress.length
                ? learningProgress[learningProgress.length - 1].performance_after.toFixed(2)
                : "--"
            }
          />
        )}
        {!isEnvOnly && (
          <StatChip
            label="Latest Primary Weakness"
            value={weaknesses.length ? weaknesses[weaknesses.length - 1].primary_weakness : "--"}
          />
        )}
        {isEnvOnly && (
          <StatChip
            label="Average Difficulty"
            value={
              environments.length
                ? (environments.reduce((sum, e) => sum + e.difficulty, 0) / environments.length).toFixed(2)
                : "--"
            }
          />
        )}
        {isDemoPpo ? (
          <StatChip label="Generalization Splits" value={generalization.length} />
        ) : (
          <StatChip label="Agent" value={isExternal ? "External" : "Not connected"} />
        )}
      </div>

      <Panel title={isEnvOnly ? "GENERATED ENVIRONMENTS" : "TRAINING HISTORY (ENVIRONMENTS)"}>
        {environments.length === 0 ? (
          <div className="text-app-mid text-xs">No environments trained yet.</div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-app-low">
                <th className="pb-2 font-medium">Environment</th>
                <th className="pb-2 font-medium">Target Weakness</th>
                <th className="pb-2 font-medium">Difficulty</th>
                <th className="pb-2 font-medium">Split</th>
              </tr>
            </thead>
            <tbody>
              {environments.map((env) => (
                <tr key={env.id} className="border-t border-[#1a2033]">
                  <td className="py-1.5 font-mono text-app-mid">{env.environment_key}</td>
                  <td className="py-1.5 text-app-orange">{env.target_weakness ?? "--"}</td>
                  <td className="py-1.5 font-mono text-app-hi">{env.difficulty.toFixed(2)}</td>
                  <td className="py-1.5 text-app-mid">{env.split}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      {isDemoPpo && (
        <Panel title="GENERALIZATION">
          {generalization.length === 0 ? (
            <div className="text-app-mid text-xs">Not yet evaluated.</div>
          ) : (
            <div className="grid grid-cols-3 gap-4">
              {generalization.map((g) => (
                <div key={g.id} className="p-3 rounded border border-app-border bg-app-panel2">
                  <div className="text-[11px] uppercase tracking-wider text-app-low mb-2">{g.split}</div>
                  <div className="text-xl font-bold text-app-hi">{g.success_rate.toFixed(2)}</div>
                  <div className="text-[11px] text-app-mid mt-1">
                    reward {g.average_reward.toFixed(2)} · steps {g.average_steps.toFixed(1)} · collisions{" "}
                    {g.collision_rate.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}
    </Shell>
  );
}
