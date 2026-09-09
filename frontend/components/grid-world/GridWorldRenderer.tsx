"use client";

import { useMemo } from "react";

export type Cell = [number, number];

export interface GridWorldRendererProps {
  gridSize: number;
  obstacles: Cell[];
  start: Cell;
  goal: Cell;
  /** Current agent position (e.g. mid-animation). Defaults to `start`. */
  agentPos?: Cell;
  /** Cells the agent has already visited -- shaded lightly. */
  visited?: Cell[];
  /** A successful route to highlight as a solid line. */
  path?: Cell[];
  /** A failed/abandoned route to highlight as a dashed line. */
  failedPath?: Cell[];
  /** Cells to flag as failure states (e.g. dead-end entries, collisions). */
  failedStates?: Cell[];
  className?: string;
}

const key = (c: Cell) => `${c[0]},${c[1]}`;

/**
 * Phase 20 -- a reusable, purely-programmatic GridWorld renderer (Section
 * 36). No screenshots or raster images anywhere: every cell, obstacle,
 * agent marker, and path line is drawn as SVG/DOM computed from the props
 * passed in. Used by the Environments page (a real generated layout, via
 * the Phase 20 `/layout` endpoint) and reusable anywhere else a maze needs
 * to be shown (Live Training's grid, once per-step agent position
 * streaming exists -- see the Live Training page's documented limitation).
 *
 * Animation: this component itself is stateless and just renders whatever
 * `agentPos`/`visited` prop it's given each render -- callers animate by
 * changing those props over time (see `useAnimatedPath` below), which
 * keeps the renderer simple, testable, and reusable for both a live
 * step-by-step animation and a static "show the final state" render.
 */
export function GridWorldRenderer({
  gridSize,
  obstacles,
  start,
  goal,
  agentPos,
  visited = [],
  path = [],
  failedPath = [],
  failedStates = [],
  className = "",
}: GridWorldRendererProps) {
  const obstacleSet = useMemo(() => new Set(obstacles.map(key)), [obstacles]);
  const visitedSet = useMemo(() => new Set(visited.map(key)), [visited]);
  const failedStateSet = useMemo(() => new Set(failedStates.map(key)), [failedStates]);
  const agent = agentPos ?? start;

  const cells: JSX.Element[] = [];
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const k = key([x, y]);
      const isObstacle = obstacleSet.has(k);
      const isStart = x === start[0] && y === start[1];
      const isGoal = x === goal[0] && y === goal[1];
      const isAgent = x === agent[0] && y === agent[1];
      const isVisited = visitedSet.has(k);
      const isFailedState = failedStateSet.has(k);

      cells.push(
        <div
          key={k}
          className={`relative flex items-center justify-center border border-[#1a2033] ${
            isObstacle
              ? "bg-[#232b40]"
              : isFailedState
                ? "bg-[#3a1420]"
                : isVisited
                  ? "bg-[#1b3a5c]/60"
                  : "bg-[#0c1120]"
          }`}
          style={{ aspectRatio: "1 / 1" }}
        >
          {isObstacle && (
            <div
              className="w-full h-full opacity-70"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, #2c3550 0, #2c3550 2px, transparent 2px, transparent 6px)",
              }}
            />
          )}
          {isStart && !isAgent && (
            <div className="absolute inset-0.5 flex items-center justify-center bg-[#16a34a] text-[8px] font-bold text-white rounded-[2px]">
              S
            </div>
          )}
          {isGoal && (
            <div className="absolute inset-0.5 flex items-center justify-center bg-[#ef4444] text-[8px] font-bold text-white rounded-[2px]">
              G
            </div>
          )}
          {isAgent && (
            <div
              className="absolute w-[55%] h-[55%] rounded-full bg-[#3f8cff] ring-4 ring-[#3f8cff]/25 transition-all duration-300 ease-linear"
              style={{ zIndex: 2 }}
            />
          )}
        </div>,
      );
    }
  }

  return (
    <div className={`relative ${className}`}>
      <div className="grid" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
        {cells}
      </div>
      {(path.length > 1 || failedPath.length > 1) && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${gridSize} ${gridSize}`}
          preserveAspectRatio="none"
        >
          {path.length > 1 && (
            <polyline
              points={path.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(" ")}
              fill="none"
              stroke="#3f8cff"
              strokeWidth={gridSize * 0.02}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          {failedPath.length > 1 && (
            <polyline
              points={failedPath.map(([x, y]) => `${x + 0.5},${y + 0.5}`).join(" ")}
              fill="none"
              stroke="#ef4444"
              strokeWidth={gridSize * 0.015}
              strokeDasharray={`${gridSize * 0.04},${gridSize * 0.03}`}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </svg>
      )}
    </div>
  );
}
