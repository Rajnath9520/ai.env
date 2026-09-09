import React, { useMemo } from "react";
import { C } from "../theme.js";
import { Panel, Bar1, KV } from "../components/ui.jsx";
import { GridWorld } from "../components/GridWorld.jsx";
import { SuccessProgressChart } from "../components/charts.jsx";
import { makeGrid, weaknesses, learningProgressData } from "../lib/mockData.js";

export default function LiveTrainingPage() {
  const grid = useMemo(() => makeGrid(48291, 14, 0.27), []);
  return (
    <div className="p-5 grid grid-cols-3 gap-4">
      <Panel title="Live environment — GridWorld-042" className="col-span-2">
        <GridWorld grid={grid} cell={28} />
      </Panel>
      <div className="flex flex-col gap-4">
        <Panel title="Agent status">
          <div className="grid grid-cols-2 gap-3">
            <KV label="Episode" value="1,842" />
            <KV label="Success rate" value="74.2%" valueColor={C.green} />
            <KV label="Avg reward" value="68.4" />
            <KV label="Avg steps" value="31.7" />
          </div>
        </Panel>
        <Panel title="Current weaknesses">
          <div className="flex flex-col gap-2">
            {weaknesses.map((w) => (
              <div key={w.key}>
                <div className="flex justify-between text-[11px] mb-1" style={{ color: C.textDim }}><span>{w.label}</span><span className="font-mono">{w.value.toFixed(2)}</span></div>
                <Bar1 value={w.value} color={w.color} />
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel title="Success rate & learning progress" className="col-span-2">
        <SuccessProgressChart data={learningProgressData} />
      </Panel>
      <Panel title="Episode log">
        <div className="flex flex-col gap-2 max-h-[220px] overflow-y-auto">
          {[1842, 1841, 1840, 1839, 1838, 1837].map((ep) => (
            <div key={ep} className="flex items-center justify-between text-[12px]" style={{ borderBottom: `1px solid ${C.border}`, paddingBottom: 6 }}>
              <span className="font-mono" style={{ color: C.textDim }}>#{ep}</span>
              <span style={{ color: ep % 3 === 0 ? C.red : C.green }}>{ep % 3 === 0 ? "failed" : "success"}</span>
              <span className="font-mono" style={{ color: C.textFaint }}>{(20 + (ep % 15)).toFixed(0)} steps</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
