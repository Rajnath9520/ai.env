"use client";

import { useEffect, useState } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { GridWorldRenderer } from "@/components/grid-world/GridWorldRenderer";
import { useAnimatedPath } from "@/hooks/use-animated-path";
import { api } from "@/lib/api-client";
import { bfsShortestPath } from "@/lib/grid-pathfinding";
import type { EnvironmentLayoutRead } from "@/types/api";

/** Fetches the real, regenerated maze for one environment (Phase 20's
 * `/layout` endpoint) and renders it with GridWorldRenderer, animating an
 * agent marker along the real BFS shortest path through that maze as a
 * demonstration of the renderer's animation support.
 *
 * Labeled clearly as a shortest-path demo, not the RL agent's actual
 * trajectory -- see grid-pathfinding.ts's docstring for why.
 */
export function EnvironmentLayoutView({
  experimentId,
  environmentId,
}: {
  experimentId: number;
  environmentId: number;
}) {
  const [layout, setLayout] = useState<EnvironmentLayoutRead | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getEnvironmentLayout(experimentId, environmentId)
      .then((l) => !cancelled && setLayout(l))
      .catch(() => !cancelled && setError("Failed to load layout"));
    return () => {
      cancelled = true;
    };
  }, [experimentId, environmentId]);

  const path = layout ? bfsShortestPath(layout.grid_size, layout.obstacles, layout.agent_start, layout.goal) : null;
  const { agentPos, visited, playing, setPlaying, reset } = useAnimatedPath(path, 350);

  if (error) return <div className="text-app-red text-xs">{error}</div>;
  if (!layout) return <div className="text-app-mid text-xs">Loading layout...</div>;

  return (
    <div className="space-y-2">
      <GridWorldRenderer
        gridSize={layout.grid_size}
        obstacles={layout.obstacles}
        start={layout.agent_start}
        goal={layout.goal}
        agentPos={agentPos}
        visited={visited}
        path={path ?? []}
      />
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-app-low">
          {path ? "shortest-path demo (BFS) -- not the agent's actual trajectory" : "no path found"}
        </span>
        <div className="flex gap-1">
          <button
            onClick={() => setPlaying((p) => !p)}
            className="p-1 rounded border border-app-border text-app-mid hover:text-app-hi hover:border-app-mid"
          >
            {playing ? <Pause size={12} /> : <Play size={12} />}
          </button>
          <button
            onClick={reset}
            className="p-1 rounded border border-app-border text-app-mid hover:text-app-hi hover:border-app-mid"
          >
            <RotateCcw size={12} />
          </button>
        </div>
      </div>
    </div>
  );
}
