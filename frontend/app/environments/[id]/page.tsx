"use client";

import { useEffect, useState } from "react";
import { Shell } from "@/components/nav/Shell";
import { Panel } from "@/components/panels/Panel";
import { EnvironmentLayoutView } from "@/components/grid-world/EnvironmentLayoutView";
import { api } from "@/lib/api-client";
import type { EnvironmentRead } from "@/types/api";

export default function EnvironmentsPage({ params }: { params: { id: string } }) {
  const experimentId = Number(params.id);
  const [environments, setEnvironments] = useState<EnvironmentRead[] | null>(null);

  useEffect(() => {
    api.getEnvironments(experimentId).then(setEnvironments);
  }, [experimentId]);

  return (
    <Shell
      experimentId={experimentId}
      title="ENVIRONMENTS"
      subtitle="Every environment generated and trained on during this experiment"
    >
      {!environments ? (
        <div className="text-app-mid text-xs">Loading...</div>
      ) : environments.length === 0 ? (
        <div className="text-app-mid text-xs">No environments generated yet.</div>
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {environments.map((env) => (
            <Panel key={env.id} title={env.environment_key.toUpperCase()}>
              <div className="flex gap-4">
                <div className="w-40 shrink-0">
                  <EnvironmentLayoutView experimentId={experimentId} environmentId={env.id} />
                </div>
                <div className="flex-1 space-y-1.5 text-[11px]">
                  <Row label="Target Weakness" value={env.target_weakness ?? "--"} accent="#f97a2c" />
                  <Row label="Difficulty" value={env.difficulty.toFixed(2)} />
                  <Row label="Grid Size" value={String(env.params.grid_size ?? "--")} />
                  <Row label="Obstacle Density" value={fmt(env.params.obstacle_density)} />
                  <Row label="Dead-End Prob." value={fmt(env.params.dead_end_probability)} />
                  <Row label="Path Complexity" value={fmt(env.params.path_complexity)} />
                  <Row label="Split" value={env.split} />
                </div>
              </div>
            </Panel>
          ))}
        </div>
      )}
    </Shell>
  );
}

function fmt(v: unknown): string {
  return typeof v === "number" ? v.toFixed(2) : "--";
}

function Row({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-app-mid">{label}</span>
      <span className="font-mono" style={{ color: accent ?? "#e6ebf5" }}>
        {value}
      </span>
    </div>
  );
}
