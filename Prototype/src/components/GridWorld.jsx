import React, { useMemo } from "react";
import { Flag } from "lucide-react";
import { C } from "../theme.js";
import { makeGrid } from "../lib/mockData.js";

export function GridWorld({ grid, cell = 26, showLegend = true }) {
  const { size, obstacles, start, goal, visited } = grid;
  const cells = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const key = `${r},${c}`;
      const isStart = r === start.r && c === start.c;
      const isGoal = r === goal.r && c === goal.c;
      const isObstacle = obstacles.has(key);
      const isVisited = visited.has(key);
      const checker = (r + c) % 2 === 0;
      let bg = checker ? "#0f1521" : "#0c1119";
      let content = null;
      if (isObstacle) bg = "#26313f";
      if (isVisited && !isObstacle) bg = C.blueDim;
      if (isGoal) { bg = C.red; content = <Flag size={12} color="#fff" />; }
      if (isStart) { bg = C.green; content = <span className="text-[9px] font-bold" style={{ color: "#06231a" }}>S</span>; }
      cells.push(
        <div key={key} className="flex items-center justify-center" style={{ width: cell, height: cell, background: bg, border: `1px solid ${C.bg}`, borderRadius: 3 }}>
          {content}
        </div>
      );
    }
  }
  return (
    <div className="flex gap-4 items-start">
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${size}, ${cell}px)`, gap: 2 }}>{cells}</div>
      {showLegend && (
        <div className="flex flex-col gap-2 pt-1">
          {[
            { c: C.blue, l: "Agent" },
            { c: C.red, l: "Goal" },
            { c: "#26313f", l: "Obstacle" },
            { c: C.blueDim, l: "Visited" },
            { c: C.green, l: "Start" },
          ].map((it) => (
            <div key={it.l} className="flex items-center gap-2 text-[11px]" style={{ color: C.textDim }}>
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: it.c }} />
              {it.l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function MiniGrid({ seed, cell = 5 }) {
  const grid = useMemo(() => makeGrid(seed, 10, 0.32), [seed]);
  const { size, obstacles } = grid;
  const cells = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const checker = (r + c) % 2 === 0;
      const isObstacle = obstacles.has(`${r},${c}`);
      cells.push(
        <div key={`${r}-${c}`} style={{ width: cell, height: cell, background: isObstacle ? "#33445c" : checker ? "#151d2c" : "#101724" }} />
      );
    }
  }
  return <div style={{ display: "grid", gridTemplateColumns: `repeat(${size}, ${cell}px)`, gap: 1 }}>{cells}</div>;
}
