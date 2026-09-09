"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { Bar, StatChip, StatusBadge } from "@/components/panels/Primitives";
import { useExperimentSocket } from "@/hooks/use-experiment-socket";
import { api } from "@/lib/api-client";
import type { AgentStateRead, ExperimentRead, GeneratedEnvironmentRead, WeaknessRead } from "@/types/api";

export default function LivePage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [experiment, setExperiment] = useState<ExperimentRead | null>(null);

  useEffect(() => {
    api.getExperiment(experimentId).then(setExperiment);
    const interval = setInterval(() => {
      api.getExperiment(experimentId).then(setExperiment);
    }, 4000);
    return () => clearInterval(interval);
  }, [experimentId]);

  if (!experiment) {
    return (
      <Shell experimentId={experimentId} title="LIVE">
        <div className="text-app-mid text-xs">Loading...</div>
      </Shell>
    );
  }

  // Phase 28 (Section 15/16/17/28) -- the entire live-view UI branches on
  // agent_mode. Each branch only ever shows fields that actually exist for
  // that mode: no reward/episode/training-progress leaks into
  // environment_only or unconnected external views.
  if (experiment.agent_mode === "demo_ppo") {
    return <DemoPPOView experiment={experiment} />;
  }
  if (experiment.agent_mode === "external") {
    return <ExternalAgentView experiment={experiment} />;
  }
  return <EnvironmentGenerationView experiment={experiment} />;
}

// ---------------------------------------------------------------------------
// Mode A: Environment Only (Section 15) -- "LIVE ENVIRONMENT GENERATION".
// No reward/episode/training-loss/success-rate fields anywhere on this page.
// ---------------------------------------------------------------------------

function EnvironmentGenerationView({ experiment }: { experiment: ExperimentRead }) {
  const socketEnabled = experiment.status !== "pending" && experiment.status !== "failed";
  const { ticks, connected, closed } = useExperimentSocket(experiment.id, socketEnabled);

  const generationTicks = ticks.filter((t) => t.type != null);
  const latestSelected = [...generationTicks].reverse().find((t) => t.type === "environment_selected");
  const record = latestSelected?.record as GeneratedEnvironmentRead | undefined;
  const candidateEvents = generationTicks.filter(
    (t) => t.type === "candidate_generated" || t.type === "candidate_validated",
  ).length;
  const totalEnvironments = generationTicks.filter((t) => t.type === "environment_selected").length;

  return (
    <Shell
      experimentId={experiment.id}
      title="LIVE ENVIRONMENT GENERATION"
      subtitle={
        <span className="flex items-center gap-3">
          <StatusBadge status={experiment.status} />
          {socketEnabled && (
            <span className={connected ? "text-app-green" : "text-app-mid"}>
              {connected ? "streaming" : closed ? "stream closed" : "connecting..."}
            </span>
          )}
        </span>
      }
    >
      <div className="text-app-mid text-xs">
        Generating, validating, and analyzing environments. Environment-only mode never creates or trains an AI
        model, so there is no reward, episode, or success rate to show here.
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Panel title="GENERATION">
          <div className="space-y-2">
            <StatChip label="Status" value={experiment.status.replace(/_/g, " ")} />
            <StatChip label="Current Environment" value={record?.environment_id ?? "--"} />
            <StatChip label="Environments Generated" value={totalEnvironments} />
            <StatChip label="Candidate Events Seen" value={candidateEvents} />
          </div>
        </Panel>

        <Panel title="ENVIRONMENT PARAMETERS">
          {record ? (
            <div className="space-y-2">
              <StatChip label="Grid Size" value={`${record.grid_size} x ${record.grid_size}`} />
              <StatChip label="Obstacle Density" value={record.obstacle_density.toFixed(2)} />
              <StatChip label="Dead-End Probability" value={record.dead_end_probability.toFixed(2)} />
              <StatChip label="Path Complexity" value={record.path_complexity.toFixed(2)} />
              <StatChip label="Difficulty" value={record.difficulty.toFixed(2)} />
            </div>
          ) : (
            <div className="text-app-mid text-xs">Waiting for the first environment to be generated...</div>
          )}
        </Panel>

        <Panel title="VALIDATION">
          {record ? (
            <div className="space-y-1.5 text-xs">
              <ValidationLine label="Reachable" ok={record.metrics.reachable_cells > 0} />
              <ValidationLine label="Valid layout" ok={record.validation.is_valid} />
              {record.validation.reasons.length > 0 && (
                <div className="text-app-red mt-2">{record.validation.reasons.join("; ")}</div>
              )}
              <div className="mt-3 text-app-mid">
                <span className="text-app-low">Path length:</span> {record.metrics.path_length ?? "--"}
                <br />
                <span className="text-app-low">Reachable cells:</span> {record.metrics.reachable_cells} /{" "}
                {record.metrics.total_cells}
                <br />
                <span className="text-app-low">Dead ends:</span> {record.metrics.dead_end_cells}
                <br />
                <span className="text-app-low">Branching factor:</span> {record.metrics.branching_factor.toFixed(2)}
              </div>
            </div>
          ) : (
            <div className="text-app-mid text-xs">No environment validated yet.</div>
          )}
        </Panel>
      </div>

      <Panel title="GENERATION REASON">
        <div className="text-xs text-app-hi">
          {record?.generation_reason ?? "No environment has been selected yet."}
        </div>
      </Panel>

      <Panel title="EVENT LOG">
        {generationTicks.length === 0 ? (
          <div className="text-app-mid text-xs">Waiting for generation events...</div>
        ) : (
          <div className="space-y-1 text-[11px] font-mono max-h-64 overflow-y-auto">
            {[...generationTicks]
              .slice(-40)
              .reverse()
              .map((t, i) => (
                <div key={i} className="text-app-mid">
                  <span className="text-app-blue">{t.type}</span>
                  {t.environment_id != null && <span> {String(t.environment_id)}</span>}
                  {t.step_index != null && <span> step={t.step_index}</span>}
                </div>
              ))}
          </div>
        )}
      </Panel>
    </Shell>
  );
}

function ValidationLine({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className={ok ? "text-app-green" : "text-app-red"}>
      {ok ? "\u2713" : "\u2717"} {label}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mode B: External Agent (Section 16) -- honest connection state; this
// page can also drive reset/step itself purely as a way to demonstrate the
// same HTTP contract an external agent process would use.
// ---------------------------------------------------------------------------

function ExternalAgentView({ experiment }: { experiment: ExperimentRead }) {
  const [state, setState] = useState<AgentStateRead | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketEnabled = experiment.status !== "pending" && experiment.status !== "failed";
  const { ticks, connected, closed } = useExperimentSocket(experiment.id, socketEnabled);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      api.getAgentState(experiment.id).then((s) => !cancelled && setState(s));
    }
    poll();
    const interval = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [experiment.id]);

  async function handleReset() {
    setBusy(true);
    setError(null);
    try {
      await api.resetAgent(experiment.id);
      const s = await api.getAgentState(experiment.id);
      setState(s);
    } catch {
      setError("Failed to reset the environment.");
    } finally {
      setBusy(false);
    }
  }

  async function handleStep(action: number) {
    setBusy(true);
    setError(null);
    try {
      await api.stepAgent(experiment.id, action);
      const s = await api.getAgentState(experiment.id);
      setState(s);
    } catch {
      setError("Failed to step the environment. Try Reset first.");
    } finally {
      setBusy(false);
    }
  }

  const connectedToAgent = state?.connected ?? false;
  const weaknessEvents = ticks.filter((t) => t.type === "weakness_detected" || t.type === "next_environment_ready");

  return (
    <Shell
      experimentId={experiment.id}
      title="EXTERNAL AGENT"
      subtitle={
        <span className="flex items-center gap-3">
          <StatusBadge status={experiment.status} />
          {socketEnabled && (
            <span className={connected ? "text-app-green" : "text-app-mid"}>
              {connected ? "streaming" : closed ? "stream closed" : "connecting..."}
            </span>
          )}
        </span>
      }
    >
      <div className="text-app-mid text-xs">
        ai.env owns the environment, trajectory collection, failure analysis, and adaptive generation. The external
        agent owns its own model and calls this experiment&apos;s reset/step HTTP endpoints directly. ai.env never
        creates a model in this mode.
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Panel title="AGENT">
          <div className="space-y-2">
            <StatChip
              label="Connection"
              value={connectedToAgent ? "Connected" : "Waiting for connection"}
              color={connectedToAgent ? "#22c55e" : "#8b96ab"}
            />
            <StatChip label="Environment" value={state?.environment_id ?? "--"} />
            <StatChip label="Episode" value={state?.episode ?? "--"} />
            <StatChip label="Step" value={state?.step ?? "--"} />
          </div>
        </Panel>

        <Panel title="LAST TRANSITION">
          {state?.last_reward != null ? (
            <div className="space-y-2">
              <StatChip
                label="Reward"
                value={state.last_reward.toFixed(3)}
                color={state.last_reward >= 0 ? "#22c55e" : "#ef4444"}
              />
              <StatChip label="Terminated" value={state.last_terminated ? "Yes" : "No"} />
              <StatChip label="Truncated" value={state.last_truncated ? "Yes" : "No"} />
              <StatChip label="Trajectories Recorded" value={state.trajectories_recorded} />
            </div>
          ) : (
            <div className="text-app-mid text-xs">No trajectory data available.</div>
          )}
        </Panel>

        <Panel title="DEMO CONTROLS">
          <div className="text-[11px] text-app-mid mb-2">
            Drives the same reset/step endpoints an external agent would call.
          </div>
          <div className="flex flex-col gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={handleReset}
              className="px-3 py-1.5 rounded bg-app-panel2 border border-app-border text-app-hi text-xs hover:border-app-blue disabled:opacity-50"
            >
              Reset
            </button>
            <div className="grid grid-cols-4 gap-1.5">
              {["Up", "Down", "Left", "Right"].map((label, action) => (
                <button
                  key={label}
                  type="button"
                  disabled={busy || !connectedToAgent}
                  onClick={() => handleStep(action)}
                  className="px-2 py-1.5 rounded bg-app-panel2 border border-app-border text-app-hi text-[11px] hover:border-app-blue disabled:opacity-50"
                >
                  {label}
                </button>
              ))}
            </div>
            {error && <div className="text-app-red text-[11px] mt-1">{error}</div>}
          </div>
        </Panel>
      </div>

      <Panel title="DETECTED WEAKNESSES">
        {state?.latest_weakness ? (
          <div className="space-y-2">
            <div className="text-[13px] font-bold text-app-orange mb-2">
              {state.latest_weakness.primary_weakness.toUpperCase()}
            </div>
            <Bar label="Obstacle Avoidance" value={state.latest_weakness.obstacle_avoidance} color="#8b96ab" />
            <Bar label="Exploration" value={state.latest_weakness.exploration} color="#8b96ab" />
            <Bar label="Dead-End Recovery" value={state.latest_weakness.dead_end_recovery} color="#8b96ab" />
            <Bar label="Path Efficiency" value={state.latest_weakness.path_efficiency} color="#8b96ab" />
            <Bar label="Short-Horizon Planning" value={state.latest_weakness.short_horizon_planning} color="#8b96ab" />
            <Bar label="Long-Horizon Planning" value={state.latest_weakness.long_horizon_planning} color="#8b96ab" />
          </div>
        ) : (
          <div className="text-app-mid text-xs">
            No agent weakness data available. Generate using configured strategy or wait for trajectory analysis.
          </div>
        )}
      </Panel>

      <Panel title="ADAPTIVE GENERATION EVENTS">
        {weaknessEvents.length === 0 ? (
          <div className="text-app-mid text-xs">No adaptive generation has happened yet.</div>
        ) : (
          <div className="space-y-1 text-[11px] font-mono max-h-56 overflow-y-auto">
            {[...weaknessEvents]
              .slice(-20)
              .reverse()
              .map((t, i) => (
                <div key={i} className="text-app-mid">
                  <span className="text-app-blue">{t.type}</span>
                  {t.reason != null && <span> -- {t.reason}</span>}
                </div>
              ))}
          </div>
        )}
      </Panel>
    </Shell>
  );
}

// ---------------------------------------------------------------------------
// Mode C: Demo Agent -- PPO (Section 17) -- the original training dashboard,
// clearly labeled, unchanged in behavior.
// ---------------------------------------------------------------------------

function DemoPPOView({ experiment }: { experiment: ExperimentRead }) {
  const [latestWeakness, setLatestWeakness] = useState<WeaknessRead | null>(null);

  useEffect(() => {
    if (experiment.status !== "running" && experiment.status !== "completed") return;
    api.getWeaknesses(experiment.id).then((rows) => {
      if (rows.length) setLatestWeakness(rows[rows.length - 1]);
    });
  }, [experiment.status, experiment.id]);

  const socketEnabled = experiment.status === "running";
  const { latestTick, ticks, connected, closed } = useExperimentSocket(experiment.id, socketEnabled);

  const rewardHistory = ticks
    .filter((t) => t.training_status === "training")
    .slice(-100)
    .map((t, i) => ({ i, reward: t.reward ?? 0, successRate: t.success_rate ?? 0 }));

  return (
    <Shell
      experimentId={experiment.id}
      title="DEMO AGENT -- PPO"
      subtitle={
        <span className="flex items-center gap-3">
          <StatusBadge status={experiment.status} />
          {socketEnabled && (
            <span className={connected ? "text-app-green" : "text-app-mid"}>
              {connected ? "streaming" : closed ? "stream closed" : "connecting..."}
            </span>
          )}
        </span>
      }
    >
      <div className="text-app-mid text-xs">
        This is the built-in PPO demonstration agent, explicitly selected for this experiment. It is not
        representative of ai.env&apos;s core environment-generation product.
      </div>

      {!socketEnabled && (
        <div className="text-app-mid text-xs">
          This experiment is <span className="font-semibold">{experiment.status}</span>. Live streaming is only
          available while a run is in progress -- start it from the{" "}
          <a href={`/experiments/${experiment.id}`} className="text-app-blue">
            experiment details page
          </a>
          .
        </div>
      )}

      <div className="grid grid-cols-3 gap-4">
        <Panel title="PPO TRAINING PROGRESS">
          <div className="space-y-2">
            <StatChip label="Environment" value={latestTick?.environment_id ?? "--"} />
            <StatChip label="Episode" value={latestTick?.episode ?? "--"} />
            <StatChip
              label="Reward"
              value={latestTick?.reward != null ? latestTick.reward.toFixed(3) : "--"}
              color={latestTick && (latestTick.reward ?? 0) >= 0 ? "#22c55e" : "#ef4444"}
            />
            <StatChip
              label="Success Rate (rolling)"
              value={latestTick?.success_rate != null ? latestTick.success_rate.toFixed(2) : "--"}
            />
          </div>
        </Panel>

        <Panel title="CURRENT WEAKNESS">
          {latestTick?.primary_weakness || latestWeakness ? (
            <>
              <div className="text-[15px] font-bold text-app-orange mb-3">
                {(latestTick?.primary_weakness ?? latestWeakness?.primary_weakness ?? "--").toUpperCase()}
              </div>
              {latestWeakness && (
                <div className="space-y-2">
                  <Bar label="Obstacle Avoidance" value={latestWeakness.obstacle_avoidance} color="#8b96ab" />
                  <Bar label="Exploration" value={latestWeakness.exploration} color="#8b96ab" />
                  <Bar label="Dead-End Recovery" value={latestWeakness.dead_end_recovery} color="#8b96ab" />
                  <Bar label="Path Efficiency" value={latestWeakness.path_efficiency} color="#8b96ab" />
                  <Bar label="Short-Horizon Planning" value={latestWeakness.short_horizon_planning} color="#8b96ab" />
                  <Bar label="Long-Horizon Planning" value={latestWeakness.long_horizon_planning} color="#8b96ab" />
                </div>
              )}
            </>
          ) : (
            <div className="text-app-mid text-xs">
              No weakness signal yet (first curriculum step trains on the base environment).
            </div>
          )}
        </Panel>

        <Panel title="LAST STEP COMPLETED">
          {(() => {
            const lastStep = [...ticks].reverse().find((t) => t.training_status === "step_completed");
            if (!lastStep) return <div className="text-app-mid text-xs">No step completed yet.</div>;
            return (
              <div className="space-y-2">
                <StatChip label="Environment" value={lastStep.environment_id ?? "--"} />
                <StatChip
                  label="Success Rate"
                  value={lastStep.success_rate != null ? lastStep.success_rate.toFixed(2) : "--"}
                />
                <StatChip
                  label="Learning Progress"
                  value={lastStep.learning_progress != null ? lastStep.learning_progress.toFixed(3) : "--"}
                  color={
                    lastStep.learning_progress != null
                      ? lastStep.learning_progress >= 0
                        ? "#22c55e"
                        : "#ef4444"
                      : undefined
                  }
                />
              </div>
            );
          })()}
        </Panel>
      </div>

      <Panel title="ROLLING REWARD / SUCCESS RATE (last 100 episodes)" className="h-64">
        {rewardHistory.length === 0 ? (
          <div className="text-app-mid text-xs">Waiting for training ticks...</div>
        ) : (
          <MiniLineChart data={rewardHistory} />
        )}
      </Panel>
    </Shell>
  );
}

function MiniLineChart({ data }: { data: { i: number; reward: number; successRate: number }[] }) {
  const width = 800;
  const height = 200;
  const maxAbsReward = Math.max(1, ...data.map((d) => Math.abs(d.reward)));
  const points = data
    .map((d, idx) => {
      const x = (idx / Math.max(1, data.length - 1)) * width;
      const y = height / 2 - (d.reward / maxAbsReward) * (height / 2 - 10);
      return `${x},${y}`;
    })
    .join(" ");
  const successPoints = data
    .map((d, idx) => {
      const x = (idx / Math.max(1, data.length - 1)) * width;
      const y = height - d.successRate * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
      <line x1={0} y1={height / 2} x2={width} y2={height / 2} stroke="#1a2236" strokeWidth={1} />
      <polyline points={points} fill="none" stroke="#3f8cff" strokeWidth={2} />
      <polyline points={successPoints} fill="none" stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4,3" />
    </svg>
  );
}
