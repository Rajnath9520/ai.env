"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { api } from "@/lib/api-client";
import type { ExperimentRead, GeneralizationRead } from "@/types/api";

const SPLIT_COLORS: Record<string, string> = {
  validation: "#3f8cff",
  test: "#22c55e",
  ood: "#f97a2c",
};

export default function GeneralizationPage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [results, setResults] = useState<GeneralizationRead[] | null>(null);
  const [experiment, setExperiment] = useState<ExperimentRead | null>(null);

  useEffect(() => {
    api.getGeneralization(experimentId).then(setResults);
    api.getExperiment(experimentId).then(setExperiment);
  }, [experimentId]);

  const maxSuccess = Math.max(0.05, ...(results ?? []).map((r) => r.success_rate));

  return (
    <Shell
      experimentId={experimentId}
      title="GENERALIZATION"
      subtitle="Trained-agent performance vs. unseen and out-of-distribution environments"
    >
      {!results ? (
        <div className="text-app-mid text-xs">Loading...</div>
      ) : results.length === 0 ? (
        <div className="text-app-mid text-xs">
          {experiment?.agent_mode === "environment_only" &&
            "This experiment has agent_mode = Environment Only, so no agent was trained and there is nothing to generalize. Generalization evaluation only applies to Demo Agent -- PPO runs."}
          {experiment?.agent_mode === "external" &&
            "This experiment's agent is external -- ai.env doesn't own or evaluate its model. Generalization evaluation only applies to Demo Agent -- PPO runs."}
          {(!experiment || experiment.agent_mode === "demo_ppo") &&
            "No generalization evaluation recorded yet -- run this experiment with generalization evaluation enabled."}
        </div>
      ) : (
        <>
          <Panel title="SUCCESS RATE BY SPLIT" className="h-72">
            <div className="flex items-end justify-around h-full pb-6">
              {results.map((r) => (
                <div key={r.id} className="flex flex-col items-center gap-2 w-24">
                  <span className="text-xs font-mono text-app-hi">{r.success_rate.toFixed(2)}</span>
                  <div
                    className="w-12 rounded-t"
                    style={{
                      height: `${Math.max(4, (r.success_rate / maxSuccess) * 180)}px`,
                      background: SPLIT_COLORS[r.split] ?? "#8b96ab",
                    }}
                  />
                  <span className="text-[11px] uppercase tracking-wider text-app-mid">{r.split}</span>
                </div>
              ))}
            </div>
          </Panel>

          <div className="grid grid-cols-3 gap-4">
            {results.map((r) => (
              <Panel key={r.id} title={r.split.toUpperCase()}>
                <div className="space-y-2 text-[12px]">
                  <Row label="Success Rate" value={r.success_rate.toFixed(3)} />
                  <Row label="Average Reward" value={r.average_reward.toFixed(3)} />
                  <Row label="Average Steps" value={r.average_steps.toFixed(1)} />
                  <Row label="Collision Rate" value={r.collision_rate.toFixed(3)} />
                </div>
              </Panel>
            ))}
          </div>
        </>
      )}
    </Shell>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-app-mid">{label}</span>
      <span className="font-mono text-app-hi">{value}</span>
    </div>
  );
}
